/**
 * \name    TLB.hpp
 * 
 * \brief   Declare the structure of TLB
 * 
 * \date    May 10, 2023
 */


#ifndef _TLB_HPP_
#define _TLB_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.h"
#include "Log.h"


/** ===============================================================================================
 * \name    TLB
 * 
 * \brief   The class of translation lookaside table for translating the index to value by hash table.
 * 
 * \note    This TLB use the Least Recently Used (LRU) algorithm. The table casting to the desired
 *          type.
 * 
 * \endcond
 * ================================================================================================
 */
template<typename Key, typename Value>
class TLB
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:

    TLB(unsigned _capacity = 0) : capacity(_capacity) {}

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    /** ==================================================================
     * \name    resize
     * 
     * \brief   change TLB capacity
     * 
     * \param   new_capacity     the key of data
     * 
     * \return  true if successful change capacity
     * 
     * \endcond
     * ===================================================================
     */
    bool resize (unsigned new_capacity)
    {
        ASSERT(new_capacity > 0, "invaild capacity: " + to_string(new_capacity));

        while (table.size() > new_capacity) {
            auto evict_key = history.front().first;
            history.pop_front();
            table.erase(evict_key);
        }

        capacity = new_capacity;

        return true;
    }

    /** ==================================================================
     * \name    lookup
     * 
     * \brief   lookup PA pair by VA
     * 
     * \param   key     the key of data
     * \param   value   the reference of data container
     * 
     * \return  true if hit
     * 
     * \endcond
     * ===================================================================
     */
    virtual bool lookup (Key key, Value& value) 
    {
        auto it = table.find(key);

        if (it == table.end()) {
            return false;
        }

        history.splice(history.end(), history, it->second);
        value = it->second->second;

        return true;
    }

    bool lookup(Key key) 
    {
        return !(table.find(key) == table.end());
    }

    /** ==================================================================
     * \name    insert
     * 
     * \brief   insert the VA and PA to the table
     * 
     * \param   key     the key of data
     * \param   value   the value going to insert 
     * 
     * \return  the value be evicted by TLB
     * 
     * \endcond
     * ===================================================================
     */
    virtual Value insert (Key key, Value value) 
    {
        Value evict_value;
        if constexpr (std::is_pointer_v<Value>) {
            evict_value = nullptr;
        } else {
            evict_value = Value{};
        }

        auto it = table.find(key);
        if (it == table.end())
        {
#if (PERFECT_ACCESS)
            if (table.size() == capacity) {
                auto evict_key = history.front().first;
                evict_value = history.front().second;
                history.pop_front();
                table.erase(evict_key);
            }
#endif
            history.emplace_back(key, value);
            auto it = history.end();
            table.emplace(key, --it);

        } else {
            history.splice(history.end(), history, it->second);
            it->second->second = value;
        }

        return evict_value;
    }

    /** ==================================================================
     * \name    erase
     * 
     * \brief   erase the VA and PA from the table
     * 
     * \param   key     the key of data
     * 
     * \return  true if erase successful
     * 
     * \endcond
     * ===================================================================
     */
    bool erase (Key key) 
    {
        auto it = table.find(key);

        if (it == table.end()) return false;

        history.erase(it->second);
        table.erase(key);
        return true;
    }

    /** ==================================================================
     * \name    release
     * 
     * \brief   release the elements from the TLB according to check 
     *          function
     * 
     * \param   check_function     the function to determine the element 
     *                             to release
     * 
     * \endcond
     * ===================================================================
     */
    int release (bool (*check_function) (const Value&)) 
    {
        int release_count = 0;
        for (auto it = history.begin(); it != history.end();)
        {
            if (check_function ((*it).second))
            {
                // cout << "erase value: " << (*it).first << endl;
                release_count++;
                table.erase((*it).first);
                history.erase(it++);
            } else {
                // cout << "not erase value: " << (*it).first << endl;
                it++;
            }
        }
        
        return release_count;
    }
    
/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    unsigned capacity;
    std::list<std::pair<Key, Value>> history;
    unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator> table;
};

template<typename Key, typename Value>
using LRU_TLB = TLB<Key, Value>;

#endif
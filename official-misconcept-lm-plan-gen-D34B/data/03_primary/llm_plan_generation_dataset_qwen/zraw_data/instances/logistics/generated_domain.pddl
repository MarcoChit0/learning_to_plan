(define (domain logistics-strips)
  (:requirements :strips)
  (:predicates
    (OBJ ?obj)
    (TRUCK ?truck)
    (LOCATION ?loc)
    (AIRPLANE ?airplane)
    (CITY ?city)
    (AIRPORT ?airport)
    (at ?obj ?loc)
    (in ?obj1 ?obj2)
    (in-city ?obj ?city)
  )

  ; (:types )		; default object

  (:action load-truck
    :parameters (?obj ?truck ?loc)
    :precondition (and
      (OBJ ?obj)
      (TRUCK ?truck)
      (LOCATION ?loc)
      (at ?truck ?loc)
      (at ?obj ?loc))
    :effect (and
      (not (at ?obj ?loc))
      (in ?obj ?truck)
    )
  )

  (:action load-airplane
    :parameters (?obj ?airplane ?loc)
    :precondition (and
      (OBJ ?obj)
      (AIRPLANE ?airplane)
      (LOCATION ?loc)
      (at ?obj ?loc)
      (at ?airplane ?loc))
    :effect (and
      (not (at ?obj ?loc))
      (in ?obj ?airplane)
    )
  )

  (:action unload-truck
    :parameters (?obj ?truck ?loc)
    :precondition (and
      (OBJ ?obj)
      (TRUCK ?truck)
      (LOCATION ?loc)
      (at ?truck ?loc)
      (in ?obj ?truck))
    :effect (and
      (not (in ?obj ?truck))
      (at ?obj ?loc))
  )

  (:action unload-airplane
    :parameters (?obj ?airplane ?loc)
    :precondition (and
      (OBJ ?obj)
      (AIRPLANE ?airplane)
      (LOCATION ?loc)
      (in ?obj ?airplane)
      (at ?airplane ?loc))
    :effect (and
      (not (in ?obj ?airplane))
      (at ?obj ?loc))
  )

  (:action drive-truck
    :parameters (?truck ?loc-from ?loc-to ?city)
    :precondition (and
      (TRUCK ?truck)
      (LOCATION ?loc-from)
      (LOCATION ?loc-to)
      (CITY ?city)
      (at ?truck ?loc-from)
      (in-city ?loc-from ?city)
      (in-city ?loc-to ?city))
    :effect (and
      (not (at ?truck ?loc-from))
      (at ?truck ?loc-to)
    )
  )

  (:action fly-airplane
    :parameters (?airplane ?loc-from ?loc-to)
    :precondition (and
      (AIRPLANE ?airplane)
      (AIRPORT ?loc-from)
      (AIRPORT ?loc-to)
      (at ?airplane ?loc-from))
    :effect (and
      (not (at ?airplane ?loc-from))
      (at ?airplane ?loc-to)
    )
  )
)
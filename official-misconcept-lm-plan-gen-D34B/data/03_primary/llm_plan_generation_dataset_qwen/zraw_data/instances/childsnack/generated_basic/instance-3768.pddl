; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 983347

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child401 - child
    bread414 - bread-portion
    content293 - content-portion
    tray336 - tray
    table66 table497 table85 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray336 kitchen)
     (at_kitchen_bread bread414)
     (at_kitchen_content content293)
     (not_allergic_gluten child401)
     (waiting child401 table66)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child401)
    )
  )
)

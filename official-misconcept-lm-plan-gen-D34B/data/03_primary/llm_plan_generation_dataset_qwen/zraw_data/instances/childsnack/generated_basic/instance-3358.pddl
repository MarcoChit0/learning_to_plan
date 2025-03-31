; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 464391

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child139 child462 - child
    bread270 bread4 - bread-portion
    content214 content113 - content-portion
    tray282 tray468 - tray
    table444 table303 table460 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray282 kitchen)
     (at tray468 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_bread bread4)
     (at_kitchen_content content214)
     (at_kitchen_content content113)
     (no_gluten_bread bread4)
     (no_gluten_content content113)
     (allergic_gluten child139)
     (not_allergic_gluten child462)
     (waiting child139 table444)
     (waiting child462 table460)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child139)
     (served child462)
    )
  )
)

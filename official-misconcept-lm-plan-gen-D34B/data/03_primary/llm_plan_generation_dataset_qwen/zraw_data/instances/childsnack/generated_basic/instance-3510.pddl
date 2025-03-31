; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 756720

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child154 child197 - child
    bread272 bread300 - bread-portion
    content46 content452 - content-portion
    tray396 tray239 - tray
    table17 table11 table210 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray396 kitchen)
     (at tray239 kitchen)
     (at_kitchen_bread bread272)
     (at_kitchen_bread bread300)
     (at_kitchen_content content46)
     (at_kitchen_content content452)
     (no_gluten_bread bread300)
     (no_gluten_content content452)
     (allergic_gluten child197)
     (not_allergic_gluten child154)
     (waiting child154 table17)
     (waiting child197 table17)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child154)
     (served child197)
    )
  )
)

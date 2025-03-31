; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 908967

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child143 child376 - child
    bread42 bread189 - bread-portion
    content216 content462 - content-portion
    tray93 tray190 tray198 - tray
    table345 table384 table407 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray93 kitchen)
     (at tray190 kitchen)
     (at tray198 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_bread bread189)
     (at_kitchen_content content216)
     (at_kitchen_content content462)
     (no_gluten_bread bread42)
     (no_gluten_content content216)
     (allergic_gluten child143)
     (not_allergic_gluten child376)
     (waiting child143 table345)
     (waiting child376 table384)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child143)
     (served child376)
    )
  )
)

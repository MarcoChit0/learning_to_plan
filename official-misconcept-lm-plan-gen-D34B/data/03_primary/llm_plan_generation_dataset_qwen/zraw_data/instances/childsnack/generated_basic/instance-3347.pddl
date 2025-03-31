; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 367085

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child485 child52 - child
    bread239 bread321 - bread-portion
    content422 content99 - content-portion
    tray233 tray15 - tray
    table251 table407 table196 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray233 kitchen)
     (at tray15 kitchen)
     (at_kitchen_bread bread239)
     (at_kitchen_bread bread321)
     (at_kitchen_content content422)
     (at_kitchen_content content99)
     (no_gluten_bread bread321)
     (no_gluten_content content422)
     (allergic_gluten child52)
     (not_allergic_gluten child485)
     (waiting child485 table407)
     (waiting child52 table196)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child485)
     (served child52)
    )
  )
)

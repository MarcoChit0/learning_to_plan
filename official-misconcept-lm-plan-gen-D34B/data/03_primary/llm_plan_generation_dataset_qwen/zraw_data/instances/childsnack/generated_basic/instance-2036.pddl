; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 733146

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child236 child486 - child
    bread245 bread433 - bread-portion
    content12 content296 - content-portion
    tray496 tray118 tray465 - tray
    table256 table359 table320 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray496 kitchen)
     (at tray118 kitchen)
     (at tray465 kitchen)
     (at_kitchen_bread bread245)
     (at_kitchen_bread bread433)
     (at_kitchen_content content12)
     (at_kitchen_content content296)
     (not_allergic_gluten child236)
     (not_allergic_gluten child486)
     (waiting child236 table320)
     (waiting child486 table359)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child236)
     (served child486)
    )
  )
)

; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 20814

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 child409 - child
    bread155 bread147 - bread-portion
    content230 content371 - content-portion
    tray42 tray326 tray380 tray36 - tray
    table359 table309 table196 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray42 kitchen)
     (at tray326 kitchen)
     (at tray380 kitchen)
     (at tray36 kitchen)
     (at_kitchen_bread bread155)
     (at_kitchen_bread bread147)
     (at_kitchen_content content230)
     (at_kitchen_content content371)
     (no_gluten_bread bread147)
     (no_gluten_content content230)
     (allergic_gluten child121)
     (not_allergic_gluten child409)
     (waiting child121 table359)
     (waiting child409 table196)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child121)
     (served child409)
    )
  )
)

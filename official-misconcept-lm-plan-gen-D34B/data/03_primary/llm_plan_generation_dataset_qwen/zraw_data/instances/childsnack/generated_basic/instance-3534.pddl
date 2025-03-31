; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 504209

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 child112 - child
    bread448 bread192 - bread-portion
    content267 content412 - content-portion
    tray451 tray116 - tray
    table290 table450 table86 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray451 kitchen)
     (at tray116 kitchen)
     (at_kitchen_bread bread448)
     (at_kitchen_bread bread192)
     (at_kitchen_content content267)
     (at_kitchen_content content412)
     (not_allergic_gluten child112)
     (not_allergic_gluten child33)
     (waiting child33 table86)
     (waiting child112 table86)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child33)
     (served child112)
    )
  )
)

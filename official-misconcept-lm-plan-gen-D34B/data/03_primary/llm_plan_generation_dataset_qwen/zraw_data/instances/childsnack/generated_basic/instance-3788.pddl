; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 834265

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 - child
    bread219 - bread-portion
    content210 - content-portion
    tray308 - tray
    table281 table387 table175 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray308 kitchen)
     (at_kitchen_bread bread219)
     (at_kitchen_content content210)
     (not_allergic_gluten child194)
     (waiting child194 table281)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child194)
    )
  )
)

; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 243251

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child471 child224 - child
    bread65 bread382 - bread-portion
    content105 content291 - content-portion
    tray429 tray358 - tray
    table446 table492 table220 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray429 kitchen)
     (at tray358 kitchen)
     (at_kitchen_bread bread65)
     (at_kitchen_bread bread382)
     (at_kitchen_content content105)
     (at_kitchen_content content291)
     (not_allergic_gluten child471)
     (not_allergic_gluten child224)
     (waiting child471 table446)
     (waiting child224 table220)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child471)
     (served child224)
    )
  )
)

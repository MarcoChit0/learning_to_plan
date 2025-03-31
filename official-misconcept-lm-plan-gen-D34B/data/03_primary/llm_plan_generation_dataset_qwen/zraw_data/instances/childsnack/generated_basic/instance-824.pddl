; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 617457

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child222 - child
    bread117 - bread-portion
    content198 - content-portion
    tray465 tray11 - tray
    table351 table400 table482 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray465 kitchen)
     (at tray11 kitchen)
     (at_kitchen_bread bread117)
     (at_kitchen_content content198)
     (not_allergic_gluten child222)
     (waiting child222 table351)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child222)
    )
  )
)

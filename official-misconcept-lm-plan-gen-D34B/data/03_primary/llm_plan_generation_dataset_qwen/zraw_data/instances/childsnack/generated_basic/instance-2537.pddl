; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 276318

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child370 - child
    bread172 - bread-portion
    content285 - content-portion
    tray404 tray356 tray351 - tray
    table340 table58 table366 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray404 kitchen)
     (at tray356 kitchen)
     (at tray351 kitchen)
     (at_kitchen_bread bread172)
     (at_kitchen_content content285)
     (not_allergic_gluten child370)
     (waiting child370 table340)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child370)
    )
  )
)

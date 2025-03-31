; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 116272

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child385 - child
    bread498 - bread-portion
    content391 - content-portion
    tray32 tray427 tray497 tray219 - tray
    table447 table117 table452 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray32 kitchen)
     (at tray427 kitchen)
     (at tray497 kitchen)
     (at tray219 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_content content391)
     (not_allergic_gluten child385)
     (waiting child385 table447)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child385)
    )
  )
)

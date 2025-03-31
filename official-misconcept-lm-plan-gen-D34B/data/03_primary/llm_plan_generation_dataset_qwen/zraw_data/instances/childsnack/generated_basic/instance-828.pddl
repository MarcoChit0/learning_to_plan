; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 580528

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child287 - child
    bread493 - bread-portion
    content409 - content-portion
    tray363 tray466 - tray
    table61 table489 table128 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray466 kitchen)
     (at_kitchen_bread bread493)
     (at_kitchen_content content409)
     (not_allergic_gluten child287)
     (waiting child287 table128)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child287)
    )
  )
)

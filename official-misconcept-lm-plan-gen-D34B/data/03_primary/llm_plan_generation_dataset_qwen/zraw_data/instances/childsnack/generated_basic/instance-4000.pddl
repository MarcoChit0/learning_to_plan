; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 491754

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child394 - child
    bread94 - bread-portion
    content331 - content-portion
    tray484 tray128 tray291 - tray
    table306 table446 table383 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at tray128 kitchen)
     (at tray291 kitchen)
     (at_kitchen_bread bread94)
     (at_kitchen_content content331)
     (not_allergic_gluten child394)
     (waiting child394 table383)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child394)
    )
  )
)

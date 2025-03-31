; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 457188

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 - child
    bread42 - bread-portion
    content32 - content-portion
    tray1 - tray
    table239 table426 table360 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray1 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_content content32)
     (not_allergic_gluten child417)
     (waiting child417 table239)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child417)
    )
  )
)

; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 377669

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child434 - child
    bread73 - bread-portion
    content17 - content-portion
    tray65 tray49 tray160 tray461 - tray
    table34 table112 table263 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray65 kitchen)
     (at tray49 kitchen)
     (at tray160 kitchen)
     (at tray461 kitchen)
     (at_kitchen_bread bread73)
     (at_kitchen_content content17)
     (not_allergic_gluten child434)
     (waiting child434 table112)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child434)
    )
  )
)

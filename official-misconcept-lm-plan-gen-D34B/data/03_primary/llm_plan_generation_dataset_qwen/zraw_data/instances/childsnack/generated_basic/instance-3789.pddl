; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 516901

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 - child
    bread312 - bread-portion
    content132 - content-portion
    tray209 - tray
    table352 table52 table213 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at_kitchen_bread bread312)
     (at_kitchen_content content132)
     (not_allergic_gluten child417)
     (waiting child417 table213)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child417)
    )
  )
)

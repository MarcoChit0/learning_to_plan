; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 413504

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 - child
    bread166 - bread-portion
    content226 - content-portion
    tray6 - tray
    table322 table295 table258 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray6 kitchen)
     (at_kitchen_bread bread166)
     (at_kitchen_content content226)
     (not_allergic_gluten child407)
     (waiting child407 table258)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child407)
    )
  )
)

; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 235565

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 - child
    bread106 - bread-portion
    content317 - content-portion
    tray211 - tray
    table186 table463 table116 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray211 kitchen)
     (at_kitchen_bread bread106)
     (at_kitchen_content content317)
     (not_allergic_gluten child349)
     (waiting child349 table186)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child349)
    )
  )
)

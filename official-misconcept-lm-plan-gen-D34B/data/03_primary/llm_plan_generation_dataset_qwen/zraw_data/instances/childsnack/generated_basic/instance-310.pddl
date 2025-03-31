; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 853912

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child165 - child
    bread248 - bread-portion
    content488 - content-portion
    tray441 - tray
    table134 table70 table379 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray441 kitchen)
     (at_kitchen_bread bread248)
     (at_kitchen_content content488)
     (not_allergic_gluten child165)
     (waiting child165 table134)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child165)
    )
  )
)

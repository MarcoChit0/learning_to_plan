; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 448072

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 - child
    bread350 - bread-portion
    content419 - content-portion
    tray277 - tray
    table425 table305 table303 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at_kitchen_bread bread350)
     (at_kitchen_content content419)
     (not_allergic_gluten child38)
     (waiting child38 table303)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child38)
    )
  )
)

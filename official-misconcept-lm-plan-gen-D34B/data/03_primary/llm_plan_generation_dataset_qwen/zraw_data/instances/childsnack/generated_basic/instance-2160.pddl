; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 230631

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child448 - child
    bread281 - bread-portion
    content39 - content-portion
    tray380 - tray
    table39 table27 table405 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at_kitchen_bread bread281)
     (at_kitchen_content content39)
     (not_allergic_gluten child448)
     (waiting child448 table405)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child448)
    )
  )
)

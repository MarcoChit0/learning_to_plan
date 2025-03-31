; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 954886

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child213 - child
    bread112 - bread-portion
    content200 - content-portion
    tray415 - tray
    table472 table359 table464 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray415 kitchen)
     (at_kitchen_bread bread112)
     (at_kitchen_content content200)
     (not_allergic_gluten child213)
     (waiting child213 table472)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child213)
    )
  )
)

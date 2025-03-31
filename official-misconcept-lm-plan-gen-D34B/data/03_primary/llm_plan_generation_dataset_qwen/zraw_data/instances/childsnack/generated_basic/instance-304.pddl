; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 427003

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child320 - child
    bread226 - bread-portion
    content414 - content-portion
    tray282 - tray
    table97 table498 table426 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray282 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_content content414)
     (not_allergic_gluten child320)
     (waiting child320 table426)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child320)
    )
  )
)

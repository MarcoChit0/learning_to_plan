; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 542475

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 - child
    bread434 - bread-portion
    content484 - content-portion
    tray209 tray129 - tray
    table90 table397 table429 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at tray129 kitchen)
     (at_kitchen_bread bread434)
     (at_kitchen_content content484)
     (not_allergic_gluten child396)
     (waiting child396 table429)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child396)
    )
  )
)

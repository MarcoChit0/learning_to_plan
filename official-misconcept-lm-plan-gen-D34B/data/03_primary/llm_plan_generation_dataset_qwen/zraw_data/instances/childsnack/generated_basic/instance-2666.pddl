; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 923529

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child9 - child
    bread394 - bread-portion
    content137 - content-portion
    tray447 tray76 tray168 tray115 - tray
    table489 table322 table15 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray447 kitchen)
     (at tray76 kitchen)
     (at tray168 kitchen)
     (at tray115 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_content content137)
     (not_allergic_gluten child9)
     (waiting child9 table489)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child9)
    )
  )
)

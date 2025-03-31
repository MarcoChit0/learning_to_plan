; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 21005

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 - child
    bread54 - bread-portion
    content133 - content-portion
    tray240 tray494 tray243 - tray
    table140 table422 table29 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray240 kitchen)
     (at tray494 kitchen)
     (at tray243 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_content content133)
     (not_allergic_gluten child92)
     (waiting child92 table140)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child92)
    )
  )
)

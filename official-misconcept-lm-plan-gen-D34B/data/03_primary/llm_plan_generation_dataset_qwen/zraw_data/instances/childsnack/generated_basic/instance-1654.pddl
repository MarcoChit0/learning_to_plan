; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 228798

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child139 - child
    bread136 - bread-portion
    content173 - content-portion
    tray427 tray300 tray325 tray412 - tray
    table2 table6 table167 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray427 kitchen)
     (at tray300 kitchen)
     (at tray325 kitchen)
     (at tray412 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_content content173)
     (not_allergic_gluten child139)
     (waiting child139 table2)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child139)
    )
  )
)

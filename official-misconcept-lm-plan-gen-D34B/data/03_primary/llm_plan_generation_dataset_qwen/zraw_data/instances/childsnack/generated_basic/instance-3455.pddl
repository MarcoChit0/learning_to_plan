; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 35982

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child443 - child
    bread279 - bread-portion
    content487 - content-portion
    tray18 tray163 tray451 tray115 - tray
    table109 table395 table118 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at tray163 kitchen)
     (at tray451 kitchen)
     (at tray115 kitchen)
     (at_kitchen_bread bread279)
     (at_kitchen_content content487)
     (not_allergic_gluten child443)
     (waiting child443 table118)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child443)
    )
  )
)

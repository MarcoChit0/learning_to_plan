; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 191889

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 child279 - child
    bread467 bread115 - bread-portion
    content28 content401 - content-portion
    tray127 tray82 tray271 tray343 - tray
    table70 table479 table332 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray127 kitchen)
     (at tray82 kitchen)
     (at tray271 kitchen)
     (at tray343 kitchen)
     (at_kitchen_bread bread467)
     (at_kitchen_bread bread115)
     (at_kitchen_content content28)
     (at_kitchen_content content401)
     (not_allergic_gluten child279)
     (not_allergic_gluten child372)
     (waiting child372 table70)
     (waiting child279 table332)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child372)
     (served child279)
    )
  )
)

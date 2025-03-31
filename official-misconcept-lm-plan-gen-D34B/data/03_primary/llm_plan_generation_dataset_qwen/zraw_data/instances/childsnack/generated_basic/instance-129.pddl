; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 849387

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child175 child453 - child
    bread309 bread260 - bread-portion
    content161 content224 - content-portion
    tray241 tray400 tray59 tray405 - tray
    table465 table161 table79 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at tray400 kitchen)
     (at tray59 kitchen)
     (at tray405 kitchen)
     (at_kitchen_bread bread309)
     (at_kitchen_bread bread260)
     (at_kitchen_content content161)
     (at_kitchen_content content224)
     (not_allergic_gluten child175)
     (not_allergic_gluten child453)
     (waiting child175 table161)
     (waiting child453 table161)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child175)
     (served child453)
    )
  )
)

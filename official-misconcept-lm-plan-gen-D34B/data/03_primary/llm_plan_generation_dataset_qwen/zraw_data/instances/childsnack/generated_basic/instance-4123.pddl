; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 332439

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child247 child47 - child
    bread374 bread488 - bread-portion
    content207 content139 - content-portion
    tray207 tray460 tray17 - tray
    table359 table51 table133 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at tray460 kitchen)
     (at tray17 kitchen)
     (at_kitchen_bread bread374)
     (at_kitchen_bread bread488)
     (at_kitchen_content content207)
     (at_kitchen_content content139)
     (not_allergic_gluten child47)
     (not_allergic_gluten child247)
     (waiting child247 table133)
     (waiting child47 table51)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child247)
     (served child47)
    )
  )
)

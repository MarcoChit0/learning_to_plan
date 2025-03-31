; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 69420

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 child171 - child
    bread104 bread321 - bread-portion
    content90 content187 - content-portion
    tray37 tray434 tray404 tray157 - tray
    table313 table284 table194 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray37 kitchen)
     (at tray434 kitchen)
     (at tray404 kitchen)
     (at tray157 kitchen)
     (at_kitchen_bread bread104)
     (at_kitchen_bread bread321)
     (at_kitchen_content content90)
     (at_kitchen_content content187)
     (not_allergic_gluten child171)
     (not_allergic_gluten child313)
     (waiting child313 table194)
     (waiting child171 table284)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child313)
     (served child171)
    )
  )
)

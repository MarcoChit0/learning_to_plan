; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 139626

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child419 child284 - child
    bread105 bread15 - bread-portion
    content474 content186 - content-portion
    tray418 tray355 tray139 - tray
    table170 table142 table157 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at tray355 kitchen)
     (at tray139 kitchen)
     (at_kitchen_bread bread105)
     (at_kitchen_bread bread15)
     (at_kitchen_content content474)
     (at_kitchen_content content186)
     (no_gluten_bread bread15)
     (no_gluten_content content186)
     (allergic_gluten child284)
     (not_allergic_gluten child419)
     (waiting child419 table142)
     (waiting child284 table170)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child419)
     (served child284)
    )
  )
)

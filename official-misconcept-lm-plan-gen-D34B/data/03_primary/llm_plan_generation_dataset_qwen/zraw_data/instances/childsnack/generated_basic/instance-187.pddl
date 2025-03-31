; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 393700

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child341 child158 - child
    bread94 bread402 - bread-portion
    content236 content325 - content-portion
    tray191 tray329 tray347 - tray
    table290 table310 table116 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray329 kitchen)
     (at tray347 kitchen)
     (at_kitchen_bread bread94)
     (at_kitchen_bread bread402)
     (at_kitchen_content content236)
     (at_kitchen_content content325)
     (not_allergic_gluten child341)
     (not_allergic_gluten child158)
     (waiting child341 table116)
     (waiting child158 table310)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child341)
     (served child158)
    )
  )
)

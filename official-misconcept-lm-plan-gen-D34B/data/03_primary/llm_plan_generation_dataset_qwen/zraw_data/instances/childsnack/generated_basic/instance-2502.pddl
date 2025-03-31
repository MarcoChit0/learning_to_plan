; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 784916

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child332 child133 - child
    bread130 bread372 - bread-portion
    content11 content343 - content-portion
    tray167 tray86 - tray
    table213 table371 table183 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at tray86 kitchen)
     (at_kitchen_bread bread130)
     (at_kitchen_bread bread372)
     (at_kitchen_content content11)
     (at_kitchen_content content343)
     (not_allergic_gluten child332)
     (not_allergic_gluten child133)
     (waiting child332 table213)
     (waiting child133 table213)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child332)
     (served child133)
    )
  )
)

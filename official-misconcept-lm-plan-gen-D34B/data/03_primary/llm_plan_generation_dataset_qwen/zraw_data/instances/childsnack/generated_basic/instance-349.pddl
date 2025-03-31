; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 6903

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child297 - child
    bread133 - bread-portion
    content268 - content-portion
    tray406 tray163 tray384 tray164 - tray
    table423 table87 table402 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray406 kitchen)
     (at tray163 kitchen)
     (at tray384 kitchen)
     (at tray164 kitchen)
     (at_kitchen_bread bread133)
     (at_kitchen_content content268)
     (not_allergic_gluten child297)
     (waiting child297 table402)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child297)
    )
  )
)

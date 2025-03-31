; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 207784

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child499 child295 - child
    bread247 bread342 - bread-portion
    content444 content322 - content-portion
    tray384 tray157 tray366 - tray
    table31 table323 table103 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at tray157 kitchen)
     (at tray366 kitchen)
     (at_kitchen_bread bread247)
     (at_kitchen_bread bread342)
     (at_kitchen_content content444)
     (at_kitchen_content content322)
     (not_allergic_gluten child499)
     (not_allergic_gluten child295)
     (waiting child499 table31)
     (waiting child295 table103)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child499)
     (served child295)
    )
  )
)

; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 459073

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child470 child73 child274 - child
    bread405 bread180 bread386 - bread-portion
    content454 content483 content163 - content-portion
    tray25 tray200 tray207 - tray
    table46 table204 table161 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray25 kitchen)
     (at tray200 kitchen)
     (at tray207 kitchen)
     (at_kitchen_bread bread405)
     (at_kitchen_bread bread180)
     (at_kitchen_bread bread386)
     (at_kitchen_content content454)
     (at_kitchen_content content483)
     (at_kitchen_content content163)
     (not_allergic_gluten child73)
     (not_allergic_gluten child470)
     (not_allergic_gluten child274)
     (waiting child470 table204)
     (waiting child73 table204)
     (waiting child274 table46)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child470)
     (served child73)
     (served child274)
    )
  )
)

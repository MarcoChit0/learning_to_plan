; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 880126

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child235 child65 - child
    bread270 bread55 - bread-portion
    content245 content151 - content-portion
    tray229 tray395 - tray
    table221 table269 table390 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at tray395 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_bread bread55)
     (at_kitchen_content content245)
     (at_kitchen_content content151)
     (not_allergic_gluten child65)
     (not_allergic_gluten child235)
     (waiting child235 table390)
     (waiting child65 table269)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child235)
     (served child65)
    )
  )
)

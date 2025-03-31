; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 283610

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 child447 - child
    bread337 bread131 - bread-portion
    content351 content460 - content-portion
    tray160 tray25 tray68 - tray
    table162 table32 table406 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray160 kitchen)
     (at tray25 kitchen)
     (at tray68 kitchen)
     (at_kitchen_bread bread337)
     (at_kitchen_bread bread131)
     (at_kitchen_content content351)
     (at_kitchen_content content460)
     (not_allergic_gluten child142)
     (not_allergic_gluten child447)
     (waiting child142 table406)
     (waiting child447 table162)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child142)
     (served child447)
    )
  )
)

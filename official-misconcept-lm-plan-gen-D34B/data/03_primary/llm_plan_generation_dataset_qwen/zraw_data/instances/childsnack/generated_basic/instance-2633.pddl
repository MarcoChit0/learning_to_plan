; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 284645

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child134 - child
    bread269 - bread-portion
    content215 - content-portion
    tray288 tray108 tray209 tray421 - tray
    table237 table70 table273 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray288 kitchen)
     (at tray108 kitchen)
     (at tray209 kitchen)
     (at tray421 kitchen)
     (at_kitchen_bread bread269)
     (at_kitchen_content content215)
     (not_allergic_gluten child134)
     (waiting child134 table70)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child134)
    )
  )
)

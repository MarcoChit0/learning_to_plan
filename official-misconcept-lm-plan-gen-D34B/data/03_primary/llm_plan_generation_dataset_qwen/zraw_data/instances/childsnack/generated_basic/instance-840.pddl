; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 118145

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child145 child118 - child
    bread114 bread461 - bread-portion
    content414 content441 - content-portion
    tray300 tray28 - tray
    table149 table464 table425 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray300 kitchen)
     (at tray28 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_bread bread461)
     (at_kitchen_content content414)
     (at_kitchen_content content441)
     (no_gluten_bread bread114)
     (no_gluten_content content414)
     (allergic_gluten child118)
     (not_allergic_gluten child145)
     (waiting child145 table149)
     (waiting child118 table149)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child145)
     (served child118)
    )
  )
)

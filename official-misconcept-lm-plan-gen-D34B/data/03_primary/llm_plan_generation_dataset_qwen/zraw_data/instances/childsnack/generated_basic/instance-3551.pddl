; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 498958

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child148 child381 - child
    bread93 bread364 - bread-portion
    content350 content348 - content-portion
    tray303 tray40 - tray
    table46 table158 table384 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray303 kitchen)
     (at tray40 kitchen)
     (at_kitchen_bread bread93)
     (at_kitchen_bread bread364)
     (at_kitchen_content content350)
     (at_kitchen_content content348)
     (not_allergic_gluten child148)
     (not_allergic_gluten child381)
     (waiting child148 table46)
     (waiting child381 table158)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child148)
     (served child381)
    )
  )
)

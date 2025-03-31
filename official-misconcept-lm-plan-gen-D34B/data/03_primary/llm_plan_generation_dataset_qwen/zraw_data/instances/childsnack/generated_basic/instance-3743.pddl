; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 122095

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child451 - child
    bread185 - bread-portion
    content222 - content-portion
    tray209 tray384 tray45 - tray
    table14 table93 table322 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at tray384 kitchen)
     (at tray45 kitchen)
     (at_kitchen_bread bread185)
     (at_kitchen_content content222)
     (not_allergic_gluten child451)
     (waiting child451 table322)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child451)
    )
  )
)

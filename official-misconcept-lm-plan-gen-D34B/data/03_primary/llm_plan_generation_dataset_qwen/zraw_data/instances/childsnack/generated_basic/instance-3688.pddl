; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 26003

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child191 child96 - child
    bread51 bread277 - bread-portion
    content252 content65 - content-portion
    tray48 tray101 - tray
    table190 table148 table318 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray48 kitchen)
     (at tray101 kitchen)
     (at_kitchen_bread bread51)
     (at_kitchen_bread bread277)
     (at_kitchen_content content252)
     (at_kitchen_content content65)
     (no_gluten_bread bread51)
     (no_gluten_content content65)
     (allergic_gluten child96)
     (not_allergic_gluten child191)
     (waiting child191 table148)
     (waiting child96 table148)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child191)
     (served child96)
    )
  )
)

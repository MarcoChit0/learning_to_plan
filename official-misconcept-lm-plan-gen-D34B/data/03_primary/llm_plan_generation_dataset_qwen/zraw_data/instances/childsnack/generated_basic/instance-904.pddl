; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 235106

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child29 child6 child220 - child
    bread422 bread47 bread157 - bread-portion
    content260 content63 content166 - content-portion
    tray385 - tray
    table82 table65 table93 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray385 kitchen)
     (at_kitchen_bread bread422)
     (at_kitchen_bread bread47)
     (at_kitchen_bread bread157)
     (at_kitchen_content content260)
     (at_kitchen_content content63)
     (at_kitchen_content content166)
     (not_allergic_gluten child6)
     (not_allergic_gluten child29)
     (not_allergic_gluten child220)
     (waiting child29 table93)
     (waiting child6 table93)
     (waiting child220 table65)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child29)
     (served child6)
     (served child220)
    )
  )
)

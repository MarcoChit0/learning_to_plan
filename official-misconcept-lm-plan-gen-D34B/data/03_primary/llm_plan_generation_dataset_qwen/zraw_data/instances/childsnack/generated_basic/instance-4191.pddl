; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 497464

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child43 child103 - child
    bread191 bread189 - bread-portion
    content333 content224 - content-portion
    tray370 - tray
    table365 table307 table354 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray370 kitchen)
     (at_kitchen_bread bread191)
     (at_kitchen_bread bread189)
     (at_kitchen_content content333)
     (at_kitchen_content content224)
     (not_allergic_gluten child103)
     (not_allergic_gluten child43)
     (waiting child43 table365)
     (waiting child103 table365)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child43)
     (served child103)
    )
  )
)

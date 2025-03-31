; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 716067

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child157 child95 child14 - child
    bread277 bread497 bread416 - bread-portion
    content276 content58 content144 - content-portion
    tray201 - tray
    table254 table43 table389 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray201 kitchen)
     (at_kitchen_bread bread277)
     (at_kitchen_bread bread497)
     (at_kitchen_bread bread416)
     (at_kitchen_content content276)
     (at_kitchen_content content58)
     (at_kitchen_content content144)
     (not_allergic_gluten child95)
     (not_allergic_gluten child14)
     (not_allergic_gluten child157)
     (waiting child157 table43)
     (waiting child95 table43)
     (waiting child14 table389)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child157)
     (served child95)
     (served child14)
    )
  )
)

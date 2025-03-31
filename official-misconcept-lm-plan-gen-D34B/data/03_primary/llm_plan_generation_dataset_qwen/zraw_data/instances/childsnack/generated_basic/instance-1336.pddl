; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 770118

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child177 child157 - child
    bread221 bread119 - bread-portion
    content88 content376 - content-portion
    tray250 tray263 tray209 - tray
    table129 table266 table67 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray250 kitchen)
     (at tray263 kitchen)
     (at tray209 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_bread bread119)
     (at_kitchen_content content88)
     (at_kitchen_content content376)
     (not_allergic_gluten child157)
     (not_allergic_gluten child177)
     (waiting child177 table266)
     (waiting child157 table67)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child177)
     (served child157)
    )
  )
)

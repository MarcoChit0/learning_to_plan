; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 12798

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child59 child184 child461 - child
    bread169 bread79 bread112 - bread-portion
    content204 content185 content270 - content-portion
    tray105 - tray
    table66 table317 table336 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at_kitchen_bread bread169)
     (at_kitchen_bread bread79)
     (at_kitchen_bread bread112)
     (at_kitchen_content content204)
     (at_kitchen_content content185)
     (at_kitchen_content content270)
     (not_allergic_gluten child184)
     (not_allergic_gluten child461)
     (not_allergic_gluten child59)
     (waiting child59 table317)
     (waiting child184 table317)
     (waiting child461 table66)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child59)
     (served child184)
     (served child461)
    )
  )
)

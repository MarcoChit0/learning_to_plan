; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 224989

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child461 child389 - child
    bread150 bread82 - bread-portion
    content17 content242 - content-portion
    tray266 tray252 - tray
    table167 table288 table355 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray266 kitchen)
     (at tray252 kitchen)
     (at_kitchen_bread bread150)
     (at_kitchen_bread bread82)
     (at_kitchen_content content17)
     (at_kitchen_content content242)
     (not_allergic_gluten child389)
     (not_allergic_gluten child461)
     (waiting child461 table355)
     (waiting child389 table355)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child461)
     (served child389)
    )
  )
)
